-- purpose: confidence distribution per match type in 0.05 buckets (threshold tuning — directly informs config changes)
-- param: hours=168
SELECT event_type,
       ROUND(CAST(confidence * 20 AS INT) / 20.0, 2) AS bucket,
       COUNT(*) AS n
FROM events
WHERE event_type IN ('face_match', 'reid_match')
  AND confidence IS NOT NULL
  AND timestamp >= datetime('now', '-{{hours}} hours')
GROUP BY event_type, bucket
ORDER BY event_type, bucket;
