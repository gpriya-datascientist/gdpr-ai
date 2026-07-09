"""
Layer: INTERFACES
Imports allowed: all layers
Purpose: FastAPI routes — thin layer, delegates everything to orchestrator.
"""
import logging
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config.container import build_document_parser, build_chunker, build_orchestrator, build_rag_service, get_settings
from domain.exceptions import EuroSecError, PIILeakError, PromptInjectionError
from domain.models import Query

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    text: str
    session_id: str | None = None
    document_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    route_taken: str
    provider_used: str
    pii_masked_count: int
    latency_ms: float
    audit_id: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


class GDPRErasureResponse(BaseModel):
    session_id: str
    entries_deleted: int
    message: str


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    settings = get_settings()
    orchestrator = build_orchestrator()
    query_obj = Query(
        raw_text=request.text,
        session_id=request.session_id,
        document_id=UUID(request.document_id) if request.document_id else None,
    )
    try:
        result = orchestrator.process(query_obj)
        return QueryResponse(
            answer=result.answer,
            route_taken=result.route_taken.value,
            provider_used=result.provider_used,
            pii_masked_count=result.pii_masked_count,
            latency_ms=round(result.latency_ms, 2),
            audit_id=str(result.audit_id),
        )
    except PromptInjectionError as e:
        logger.error("Injection attempt blocked: %s", e)
        raise HTTPException(status_code=400, detail="Query blocked: security violation")
    except EuroSecError as e:
        logger.error("EuroSecError: %s", e)
        raise HTTPException(status_code=500, detail="Internal processing error")
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    import tempfile, os, csv, io, json
    from domain.models import Document
    from uuid import uuid4

    settings = get_settings()
    suffix = os.path.splitext(file.filename)[1].lower()
    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── CSV handler (METRIS sensor logs) ──────────────────────
        if suffix == ".csv":
            text_content = content.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text_content))
            rows = list(reader)
            total_rows = len(rows)
            if total_rows == 0:
                raise HTTPException(status_code=400, detail="CSV file is empty")

            anomaly_total = sum(1 for r in rows if str(r.get('is_anomaly','0')).strip()=='1')
            defect_total  = sum(1 for r in rows if str(r.get('defect_flag','0')).strip()=='1')
            fault_rows    = [r for r in rows if str(r.get('machine_status',''))=='FAULT']
            warning_rows  = [r for r in rows if str(r.get('machine_status',''))=='WARNING']

            parts = {}
            for r in rows:
                p = r.get('part','unknown')
                parts[p] = parts.get(p,0) + 1

            last_row = rows[-1]
            total_door_left  = last_row.get('total_door_left','N/A')
            total_door_right = last_row.get('total_door_right','N/A')
            total_hood       = last_row.get('total_hood','N/A')
            after_each_step  = last_row.get('after_each_step','N/A')

            stroke_stats = {}
            for part in parts:
                vals = [float(r['stroke_rate']) for r in rows if r.get('part')==part and r.get('stroke_rate','')]
                if vals:
                    stroke_stats[part] = {'avg':round(sum(vals)/len(vals),2),'min':round(min(vals),2),'max':round(max(vals),2)}

            cycle_times = []
            for r in rows:
                try: cycle_times.append(float(r['cycle_time_s']))
                except: pass
            avg_cycle = round(sum(cycle_times)/len(cycle_times),1) if cycle_times else 'N/A'

            anomaly_by_day = {}
            for r in rows:
                if str(r.get('is_anomaly','0')).strip()=='1':
                    day = str(r.get('timestamp',''))[:10]
                    anomaly_by_day[day] = anomaly_by_day.get(day,0) + 1

            anomaly_by_part = {}
            for r in rows:
                if str(r.get('is_anomaly','0')).strip()=='1':
                    p = r.get('part','unknown')
                    anomaly_by_part[p] = anomaly_by_part.get(p,0) + 1

            worst_day = max(anomaly_by_day, key=anomaly_by_day.get) if anomaly_by_day else 'none'
            worst_part = max(anomaly_by_part, key=anomaly_by_part.get) if anomaly_by_part else 'none'

            summary_chunk = f"""METRIS SENSOR LOG — COMPLETE STATISTICS
File: {file.filename}
Date Range: {rows[0].get('timestamp','?')} to {rows[-1].get('timestamp','?')}
Total Records: {total_rows}

=== PRODUCTION TOTALS ===
Door Left parts produced: {total_door_left}
Door Right parts produced: {total_door_right}
Hood parts produced: {total_hood}
After each step total: {after_each_step}
Readings per part: {', '.join([f'{k}={v}' for k,v in parts.items()])}

=== ANOMALY STATISTICS ===
Total anomalies: {anomaly_total} ({anomaly_total/max(total_rows,1)*100:.1f}%)
Total defects: {defect_total} ({defect_total/max(total_rows,1)*100:.1f}%)
Total FAULT events: {len(fault_rows)}
Total WARNING events: {len(warning_rows)}
Day with most anomalies: {worst_day} ({anomaly_by_day.get(worst_day,0)} anomalies)
Part with most anomalies: {worst_part} ({anomaly_by_part.get(worst_part,0)} anomalies)
Anomalies by day: {', '.join([f'{k}={v}' for k,v in sorted(anomaly_by_day.items())])}
Anomalies by part: {', '.join([f'{k}={v}' for k,v in anomaly_by_part.items()])}

=== STROKE RATE STATISTICS ===
{chr(10).join([f'{p}: avg={s["avg"]}, min={s["min"]}, max={s["max"]}' for p,s in stroke_stats.items()])}

=== CYCLE TIME STATISTICS ===
Average cycle time: {avg_cycle} seconds
Target cycle time: 160 seconds (for 50% speed improvement)

=== MACHINE STATUS ===
RUNNING readings: {sum(1 for r in rows if r.get('machine_status')=='RUNNING')}
WARNING readings: {len(warning_rows)}
FAULT readings: {len(fault_rows)}"""

            chunks = [summary_chunk]
            days = {}
            for r in rows:
                day = str(r.get('timestamp',''))[:10]
                if day not in days: days[day] = []
                days[day].append(r)

            for day, day_rows in sorted(days.items()):
                day_anomalies = sum(1 for r in day_rows if str(r.get('is_anomaly','0')).strip()=='1')
                day_defects   = sum(1 for r in day_rows if str(r.get('defect_flag','0')).strip()=='1')
                day_faults    = [r for r in day_rows if r.get('machine_status')=='FAULT']
                day_strokes   = [float(r['stroke_rate']) for r in day_rows if r.get('stroke_rate','')]
                avg_stroke    = round(sum(day_strokes)/len(day_strokes),2) if day_strokes else 0
                chunks.append(f"""METRIS Daily Report: {day}
Total readings: {len(day_rows)} | Anomalies: {day_anomalies} | Defects: {day_defects} | Faults: {len(day_faults)}
Average stroke rate: {avg_stroke}
Parts: {', '.join(set(r.get('part','') for r in day_rows))}
Fault events: {'; '.join([f"tool={r.get('current_tool')} part={r.get('part')} stroke={r.get('stroke_rate')}" for r in day_faults[:3]])}""")

            doc = Document(id=uuid4(), filename=file.filename, content=summary_chunk, chunks=chunks)
            rag = build_rag_service(settings)
            rag.ingest_document(doc, chunks)
            return UploadResponse(document_id=str(doc.id), filename=file.filename,
                                  chunks_created=len(chunks),
                                  message=f"METRIS CSV: {total_rows} rows → {len(chunks)} chunks")

        # ── JSON handler (daily summary) ──────────────────────────
        if suffix == ".json":
            text_content = content.decode("utf-8", errors="replace")
            data = json.loads(text_content)

            if isinstance(data, list) and len(data) > 0:
                total_anomalies = sum(d.get('anomalies',0) for d in data)
                total_defects   = sum(d.get('defects',0) for d in data)
                total_faults    = sum(d.get('faults',0) for d in data)
                worst_day       = max(data, key=lambda d: d.get('anomalies',0))
                best_day        = min(data, key=lambda d: d.get('anomalies',0))
                avg_stroke      = round(sum(d.get('avg_stroke_rate',0) for d in data)/max(len(data),1),2)
                avg_cycle       = round(sum(d.get('avg_cycle_time',0) for d in data)/max(len(data),1),1)

                summary = f"""METRIS DAILY SUMMARY — COMPLETE REPORT
File: {file.filename}
Total days: {len(data)}
Date range: {data[0].get('date','?')} to {data[-1].get('date','?')}

=== OVERALL TOTALS ===
Total anomalies across all days: {total_anomalies}
Total defects across all days: {total_defects}
Total FAULT events: {total_faults}
Average daily anomalies: {round(total_anomalies/max(len(data),1),1)}

=== BEST / WORST DAYS ===
Worst day: {worst_day.get('date')} with {worst_day.get('anomalies')} anomalies
Best day: {best_day.get('date')} with {best_day.get('anomalies')} anomalies

=== AVERAGES ===
Average stroke rate: {avg_stroke}
Average cycle time: {avg_cycle} seconds
Target cycle time: 160 seconds

=== ALL DAILY DATA ===
{chr(10).join([f"Date:{d.get('date')} Anomalies:{d.get('anomalies',0)} Defects:{d.get('defects',0)} Faults:{d.get('faults',0)} AvgStroke:{d.get('avg_stroke_rate',0)} AvgCycle:{d.get('avg_cycle_time',0)}s DoorLeft:{d.get('parts_door_left',0)} DoorRight:{d.get('parts_door_right',0)} Hood:{d.get('parts_hood',0)}" for d in data])}"""

                chunks = [summary]
                for d in data:
                    chunks.append(f"""Daily Report: {d.get('date','?')}
Anomalies: {d.get('anomalies',0)} | Defects: {d.get('defects',0)} | Faults: {d.get('faults',0)} | Readings: {d.get('total_readings',0)}
Avg stroke rate: {d.get('avg_stroke_rate',0)} | Avg cycle time: {d.get('avg_cycle_time',0)}s
Door Left: {d.get('parts_door_left',0)} | Door Right: {d.get('parts_door_right',0)} | Hood: {d.get('parts_hood',0)}""")
            else:
                chunks = [f"METRIS JSON Data:\n{json.dumps(data, indent=2)[:3000]}"]

            doc = Document(id=uuid4(), filename=file.filename, content=chunks[0], chunks=chunks)
            rag = build_rag_service(settings)
            rag.ingest_document(doc, chunks)
            return UploadResponse(document_id=str(doc.id), filename=file.filename,
                                  chunks_created=len(chunks),
                                  message=f"JSON ingested: {len(chunks)} chunks ready")

        # ── Standard handler (PDF, TXT, DOCX) ─────────────────────
        parser = build_document_parser()
        chunker = build_chunker(settings)
        rag = build_rag_service(settings)
        document = parser.parse(tmp_path)
        document.filename = file.filename
        chunks = chunker.chunk(document)
        rag.ingest_document(document, chunks)
        return UploadResponse(document_id=str(document.id), filename=file.filename,
                              chunks_created=len(chunks),
                              message="Document ingested successfully")
    finally:
        os.unlink(tmp_path)


@router.delete("/gdpr/erase/{session_id}", response_model=GDPRErasureResponse)
async def gdpr_erase(session_id: str):
    from config.container import build_audit_logger
    settings = get_settings()
    audit = build_audit_logger(settings)
    count = audit.delete_by_session(session_id)
    return GDPRErasureResponse(session_id=session_id, entries_deleted=count,
                               message=f"GDPR erasure complete: {count} records deleted")


@router.get("/audit/{session_id}")
async def get_audit_log(session_id: str):
    from config.container import build_audit_logger
    settings = get_settings()
    audit = build_audit_logger(settings)
    entries = audit.get_entries(session_id)
    return {
        "session_id": session_id,
        "total_entries": len(entries),
        "entries": [
            {"id": str(e.id), "route": e.route_decision.value,
             "sensitivity": e.sensitivity_level.value, "pii_detected": e.pii_detected,
             "provider": e.provider_called, "gdpr_compliant": e.gdpr_compliant,
             "timestamp": e.created_at.isoformat()}
            for e in entries
        ],
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "EuroSec AI",
            "local_only_mode": get_settings().cloud_provider.value == "none"}
