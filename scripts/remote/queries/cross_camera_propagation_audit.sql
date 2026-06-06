-- purpose: audit identity transfers — zone propagation events and reid->face upgrades (extra_data shows from/to)
-- param: hours=48
-- param: limit=100
SELECT e.id, e.timestamp, e.camera_id, e.track_id, p.name AS person_name,
       e.confidence, e.extra_data, e.snapshot_path
FROM events e
LEFT JOIN persons p ON p.id = e.person_id
WHERE e.event_type IN ('cross_camera_propagation', 'id_upgraded')
  AND e.timestamp >= datetime('now', '-{{hours}} hours')
ORDER BY e.timestamp DESC
LIMIT {{limit}};
