-- purpose: face/reid matches below a confidence ceiling (suspect identifications; feed ids into fetch_snapshots.sh)
-- param: hours=24
-- param: below=0.75
-- param: limit=50
SELECT e.id, e.timestamp, e.camera_id, e.event_type, e.track_id,
       p.name AS person_name, e.confidence,
       json_extract(e.extra_data, '$.threshold') AS configured_threshold,
       json_extract(e.extra_data, '$.match_policy') AS match_policy,
       e.snapshot_path
FROM events e
LEFT JOIN persons p ON p.id = e.person_id
WHERE e.event_type IN ('face_match', 'reid_match')
  AND e.timestamp >= datetime('now', '-{{hours}} hours')
  AND e.confidence < {{below}}
ORDER BY e.confidence ASC
LIMIT {{limit}};
