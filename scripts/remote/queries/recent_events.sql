-- purpose: Most recent events with person names — general triage starting point
-- param: hours=24
-- param: limit=100
SELECT e.id, datetime(e.timestamp, 'localtime') AS local_time, e.timestamp AS utc,
       e.camera_id, e.event_type, e.track_id,
       p.name AS person_name, e.confidence, e.snapshot_path, e.extra_data
FROM events e
LEFT JOIN persons p ON p.id = e.person_id
WHERE e.timestamp >= datetime('now', '-{{hours}} hours')
ORDER BY e.timestamp DESC
LIMIT {{limit}};
