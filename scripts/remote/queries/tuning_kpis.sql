-- purpose: self-tuning KPI snapshot per camera (fragmentation, identification coverage, misID signals)
-- param: hours=24
WITH created AS (
    SELECT camera_id, timestamp,
           (julianday(timestamp) - julianday(LAG(timestamp) OVER (PARTITION BY camera_id ORDER BY timestamp))) * 86400 AS gap_s
    FROM events
    WHERE event_type = 'track_created' AND timestamp >= datetime('now', '-{{hours}} hours')
),
frag AS (
    SELECT camera_id, COUNT(*) AS tracks_created, SUM(gap_s < 3) AS recreated_within_3s
    FROM created GROUP BY camera_id
),
cov AS (
    SELECT last_camera_id AS camera_id,
           COUNT(*) AS tracks,
           SUM(person_id IS NOT NULL) AS tracks_identified,
           SUM((julianday(last_seen) - julianday(first_seen)) * 86400 >= 10) AS tracks_10s,
           SUM(person_id IS NOT NULL AND (julianday(last_seen) - julianday(first_seen)) * 86400 >= 10) AS tracks_10s_identified
    FROM tracks WHERE first_seen >= datetime('now', '-{{hours}} hours')
    GROUP BY last_camera_id
),
ev AS (
    SELECT camera_id,
           SUM(event_type = 'face_match') AS face_matches,
           SUM(event_type = 'reid_match') AS reid_matches,
           SUM(event_type = 'cross_camera_propagation') AS propagations,
           SUM(event_type = 'id_upgraded') AS id_upgrades,
           SUM(event_type = 'person_zone_change') AS zone_changes,
           SUM(event_type = 'unidentified_face_saved') AS unidentified_faces
    FROM events WHERE timestamp >= datetime('now', '-{{hours}} hours')
    GROUP BY camera_id
),
flip AS (
    SELECT camera_id, COUNT(*) AS flipflop_tracks FROM (
        SELECT track_id, MIN(camera_id) AS camera_id
        FROM events
        WHERE event_type IN ('face_match', 'reid_match', 'id_upgraded')
          AND person_id IS NOT NULL AND track_id IS NOT NULL
          AND timestamp >= datetime('now', '-{{hours}} hours')
        GROUP BY track_id HAVING COUNT(DISTINCT person_id) > 1
    ) GROUP BY camera_id
)
SELECT f.camera_id,
       f.tracks_created,
       f.recreated_within_3s,
       ROUND(1.0 * f.recreated_within_3s / f.tracks_created, 3) AS recreate_rate,
       c.tracks_10s,
       c.tracks_10s_identified,
       ROUND(1.0 * c.tracks_10s_identified / NULLIF(c.tracks_10s, 0), 3) AS coverage_10s,
       e.face_matches, e.reid_matches, e.propagations, e.id_upgrades,
       e.zone_changes, e.unidentified_faces,
       COALESCE(fl.flipflop_tracks, 0) AS flipflop_tracks
FROM frag f
LEFT JOIN cov c ON c.camera_id = f.camera_id
LEFT JOIN ev e ON e.camera_id = f.camera_id
LEFT JOIN flip fl ON fl.camera_id = f.camera_id
ORDER BY f.camera_id;
