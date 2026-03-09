CREATE TABLE IF NOT EXISTS workspace.default.brand_detection_bronze
USING DELTA AS
SELECT * FROM read_files(
  '/Volumes/workspace/default/brand_detection_raw/brand_data/',
  format => 'csv',
  header => 'true',
  inferSchema => 'true'
);
