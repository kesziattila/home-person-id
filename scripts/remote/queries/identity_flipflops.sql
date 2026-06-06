-- purpose: tracks assigned more than one distinct person (core misidentification signal; drill down with track_events)
-- param: hours=48
SELECT e.track_id,
       COUNT(DISTINCT e.person_id) AS distinct_persons,
       GROUP_CONCAT(DISTINCT p.name) AS person_names,
       MIN(e.timestamp) AS first_event,
       MAX(e.timestamp) AS last_event,
       COUNT(*) AS match_events
FROM events e
LEFT JOIN persons p ON p.id = e.person_id
WHERE e.event_type IN ('face_match', 'reid_match', 'id_upgraded')
  AND e.person_id IS NOT NULL
  AND e.track_id IS NOT NULL
  AND e.timestamp >= datetime('now', '-{{hours}} hours')
GROUP BY e.track_id
HAVING COUNT(DISTINCT e.person_id) > 1
ORDER BY distinct_persons DESC, last_event DESC;
