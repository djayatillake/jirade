{{
  auto_config(
    materialized='table',
    databricks_tags={'domain': 'sales', 'sub_domain': 'opportunities'}
  )
}}

-- New sales signal fact table: tracks AE-touched accounts that progressed
-- through pipeline this quarter.
SELECT
  o.opportunity_id,
  o.account_id,
  o.stage,
  o.close_date,
  a.account_owner_name
FROM {{ ref('mart__sales__fact_opportunity') }} o
LEFT JOIN {{ ref('analytics__dimensional__dim_account') }} a
  ON o.account_id = a.account_id
