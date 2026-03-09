CREATE OR REPLACE TABLE workspace.default.brand_detection_gold
USING DELTA AS
SELECT
  category,
  COUNT(*) AS total_items,
  ROUND(AVG(confidence_score),4) AS avg_confidence,
  ROUND(MIN(confidence_score),4) AS min_confidence,
  ROUND(MAX(confidence_score),4) AS max_confidence,
  SUM(CASE WHEN confidence_tier='high'   THEN 1 ELSE 0 END) AS high_confidence_count,
  SUM(CASE WHEN confidence_tier='medium' THEN 1 ELSE 0 END) AS medium_confidence_count,
  SUM(CASE WHEN confidence_tier='low'    THEN 1 ELSE 0 END) AS low_confidence_count,
  ROUND(
    SUM(CASE WHEN confidence_tier='high' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2
  ) AS high_confidence_pct,
  CURRENT_TIMESTAMP() AS gold_processed_at
FROM workspace.default.brand_detection_silver
GROUP BY category
ORDER BY total_items DESC;
