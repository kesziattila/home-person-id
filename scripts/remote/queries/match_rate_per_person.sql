-- purpose: per person: face vs reid match counts + confidence stats (find a degraded gallery)
-- param: hours=168
SELECT p.name AS person_name,
       SUM(e.event_type = 'face_match') AS face_matches,
       SUM(e.event_type = 'reid_match') AS reid_matches,
       ROUND(AVG(e.confidence), 3) AS avg_conf,
       ROUND(MIN(e.confidence), 3) AS min_conf,
       ROUND(MAX(e.confidence), 3) AS max_conf,
       MAX(e.timestamp) AS last_match
FROM events e
JOIN persons p ON p.id = e.person_id
WHERE e.event_type IN ('face_match', 'reid_match')
  AND e.timestamp >= datetime('now', '-{{hours}} hours')
GROUP BY p.id
ORDER BY p.name;
