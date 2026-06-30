-- (fixture) existing dim_account — already classified, advisor should skip.
SELECT * FROM {{ source('salesforce', 'account') }}
