-- purpose: full chronological event sequence for one track (drill-down after identity_flipflops)
-- param: track_id=global_1
SELECT e.id, e.timestamp, e.camera_id, e.event_type, p.name AS person_name,
       e.confidence, e.extra_data, e.snapshot_path
FROM events e
LEFT JOIN persons p ON p.id = e.person_id
WHERE e.track_id = '{{track_id}}'
ORDER BY e.timestamp;
