-- purpose: per camera: tracks created vs face/reid matches vs unidentified saves (find a bad camera/angle)
-- param: hours=48
SELECT camera_id,
       SUM(event_type = 'track_created') AS tracks_created,
       SUM(event_type = 'face_match') AS face_matches,
       SUM(event_type = 'reid_match') AS reid_matches,
       SUM(event_type = 'unidentified_face_saved') AS unidentified_saved,
       ROUND(AVG(CASE WHEN event_type IN ('face_match', 'reid_match') THEN confidence END), 3) AS avg_match_conf
FROM events
WHERE timestamp >= datetime('now', '-{{hours}} hours')
GROUP BY camera_id
ORDER BY camera_id;
