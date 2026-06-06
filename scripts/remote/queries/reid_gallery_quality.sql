-- purpose: Re-ID gallery health per person: embedding counts, quality/visibility stats, snapshot coverage
SELECT p.name AS person_name,
       COUNT(r.id) AS embeddings,
       ROUND(AVG(r.quality), 3) AS avg_quality,
       ROUND(AVG(r.visibility), 3) AS avg_visibility,
       SUM(r.snapshot_path IS NULL) AS missing_snapshots,
       MIN(r.timestamp) AS oldest,
       MAX(r.timestamp) AS newest
FROM reid_embeddings r
JOIN persons p ON p.id = r.person_id
GROUP BY p.id
ORDER BY p.name;
