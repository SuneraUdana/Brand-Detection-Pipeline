CREATE OR REPLACE TABLE workspace.default.brand_detection_silver
USING DELTA AS
SELECT
  _unit_id                        AS item_id,
  LOWER(TRIM(category))           AS category,
  ROUND(`category:confidence`,4)  AS confidence_score,
  image_url,
  CASE
    WHEN `category:confidence` >= 0.8 THEN 'high'
    WHEN `category:confidence` >= 0.5 THEN 'medium'
    ELSE 'low'
  END                             AS confidence_tier,
  CURRENT_TIMESTAMP()             AS processed_at
FROM workspace.default.brand_detection_bronze
WHERE image_url IS NOT NULL
  AND category  IS NOT NULL;
