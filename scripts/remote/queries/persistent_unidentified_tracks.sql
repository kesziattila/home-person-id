-- purpose: long-lived tracks that never got identified (missed-match signal)
-- param: hours=48
-- param: min_seconds=30
SELECT t.id AS track_id, t.first_seen, t.last_seen,
       CAST((julianday(t.last_seen) - julianday(t.first_seen)) * 86400 AS INT) AS lifetime_sec,
       t.last_camera_id, t.status, t.extra_data
FROM tracks t
WHERE t.person_id IS NULL
  AND t.first_seen >= datetime('now', '-{{hours}} hours')
  AND (julianday(t.last_seen) - julianday(t.first_seen)) * 86400 >= {{min_seconds}}
ORDER BY lifetime_sec DESC;
