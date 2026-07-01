{{
  auto_config(materialized='table')
}}

-- Already carries a real governed domain (growth) in schema.yml.
SELECT
  a.application_id,
  a.week,
  a.active_users
FROM {{ ref('mart__growth__rpt_dashboard_application_user') }} a
