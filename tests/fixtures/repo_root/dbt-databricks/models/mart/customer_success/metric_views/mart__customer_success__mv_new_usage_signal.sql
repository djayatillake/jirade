{{
  auto_config(
    materialized='metric_view',
    databricks_tags={'domain': 'customer_success_professional_services', 'sub_domain': 'customer_success'}
  )
}}

-- New CS metric view: account-level usage rollup.
version: 1.0
source: |
  SELECT
    rcu.*,
    dac.customer_account_segment
  FROM {{ ref('mart__customer_success__rpt_current_usage') }} rcu
  LEFT JOIN {{ ref('analytics__dimensional__dim_account') }} dac
    ON rcu.account_id = dac.account_id
