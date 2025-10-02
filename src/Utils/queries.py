
all_data = '''
select
*
from
paysim
'''

dataframe_metrics = '''
    SELECT  
    COUNT(*) AS count,  
    ROUND(MIN(amount), 2) AS min_amount,  
    ROUND(MAX(amount), 2) AS max_amount,  
    ROUND(AVG(amount), 2) AS avg_amount,  
    ROUND(SUM(amount), 2) AS sum_amount,  
    MIN(step) AS min_date,  
    MAX(step) AS max_date,  
    SUM(isFraud) AS fraud_count, 
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount  
    FROM paysim
'''

time_data = '''
    select
    datetime('2025-10-01', '-' || step || ' hours') date,
    count(*) count
    from paysim
    group by step
    order by step asc
'''

transaction_type_analysis = '''
    SELECT
        type,
        COUNT(*) AS count,
        ROUND(MIN(amount), 2) AS min_amount,
        ROUND(MAX(amount), 2) AS max_amount,
        ROUND(AVG(amount), 2) AS avg_amount,
        ROUND(SUM(amount), 2) AS sum_amount,
        SUM(isFraud) AS fraud_count,
        SUM(isFlaggedFraud) AS flagged_fraud_count
    FROM paysim
    GROUP BY type
    ORDER BY count DESC
'''