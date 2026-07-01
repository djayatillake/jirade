{{
  auto_config(materialized='table')
}}

-- New growth activation signal. Tags are declared in schema.yml (domain: tbd),
-- so the tag advisor should propose a real governed domain.
SELECT
  o.organization_id,
  o.first_search_at,
  o.activated_at
FROM {{ ref('mart__growth__rpt_dashboard_organization') }} o
