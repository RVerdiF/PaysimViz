'''
Contains all queries used in the app.
'''
all_data = '''
select
*
from
paysim
'''

home_page_data = '''
    WITH ColumnCounts AS (
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN step IS NULL THEN 1 ELSE 0 END) AS step_nulls,
            SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END) AS type_nulls,
            SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS amount_nulls,
            SUM(CASE WHEN nameOrig IS NULL THEN 1 ELSE 0 END) AS nameOrig_nulls,
            SUM(CASE WHEN oldbalanceOrg IS NULL THEN 1 ELSE 0 END) AS oldbalanceOrg_nulls,
            SUM(CASE WHEN newbalanceOrig IS NULL THEN 1 ELSE 0 END) AS newbalanceOrig_nulls,
            SUM(CASE WHEN nameDest IS NULL THEN 1 ELSE 0 END) AS nameDest_nulls,
            SUM(CASE WHEN oldbalanceDest IS NULL THEN 1 ELSE 0 END) AS oldbalanceDest_nulls,
            SUM(CASE WHEN newbalanceDest IS NULL THEN 1 ELSE 0 END) AS newbalanceDest_nulls,
            SUM(CASE WHEN isFraud IS NULL THEN 1 ELSE 0 END) AS isFraud_nulls,
            SUM(CASE WHEN isFlaggedFraud IS NULL THEN 1 ELSE 0 END) AS isFlaggedFraud_nulls
        FROM paysim
    )
    SELECT * FROM ColumnCounts
'''

descriptive_stats_query = '''
    SELECT
        'count' AS metric,
        COUNT(step) AS step,
        COUNT(amount) AS amount,
        COUNT(oldbalanceOrg) AS oldbalanceOrg,
        COUNT(newbalanceOrig) AS newbalanceOrig,
        COUNT(oldbalanceDest) AS oldbalanceDest,
        COUNT(newbalanceDest) AS newbalanceDest,
        COUNT(isFraud) AS isFraud,
        COUNT(isFlaggedFraud) AS isFlaggedFraud
    FROM paysim
    UNION ALL
    SELECT
        'mean' AS metric,
        AVG(step) AS step,
        AVG(amount) AS amount,
        AVG(oldbalanceOrg) AS oldbalanceOrg,
        AVG(newbalanceOrig) AS newbalanceOrig,
        AVG(oldbalanceDest) AS oldbalanceDest,
        AVG(newbalanceDest) AS newbalanceDest,
        AVG(isFraud) AS isFraud,
        AVG(isFlaggedFraud) AS isFlaggedFraud
    FROM paysim
    UNION ALL
    SELECT
        'min' AS metric,
        MIN(step) AS step,
        MIN(amount) AS amount,
        MIN(oldbalanceOrg) AS oldbalanceOrg,
        MIN(newbalanceOrig) AS newbalanceOrig,
        MIN(oldbalanceDest) AS oldbalanceDest,
        MIN(newbalanceDest) AS newbalanceDest,
        MIN(isFraud) AS isFraud,
        MIN(isFlaggedFraud) AS isFlaggedFraud
    FROM paysim
    UNION ALL
    SELECT
        'max' AS metric,
        MAX(step) AS step,
        MAX(amount) AS amount,
        MAX(oldbalanceOrg) AS oldbalanceOrg,
        MAX(newbalanceOrig) AS newbalanceOrig,
        MAX(oldbalanceDest) AS oldbalanceDest,
        MAX(newbalanceDest) AS newbalanceDest,
        MAX(isFraud) AS isFraud,
        MAX(isFlaggedFraud) AS isFlaggedFraud
    FROM paysim
'''

zero_amount_by_type_query = '''
    SELECT type, COUNT(*) as count FROM paysim WHERE amount = 0 GROUP BY type
'''

zero_amount_by_fraud_query = '''
    SELECT isFraud, COUNT(*) as count FROM paysim WHERE amount = 0 GROUP BY isFraud
'''

negative_value_query = '''
    SELECT
        SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) as amount,
        SUM(CASE WHEN oldbalanceOrg < 0 THEN 1 ELSE 0 END) as oldbalanceOrg,
        SUM(CASE WHEN newbalanceOrig < 0 THEN 1 ELSE 0 END) as newbalanceOrig,
        SUM(CASE WHEN oldbalanceDest < 0 THEN 1 ELSE 0 END) as oldbalanceDest,
        SUM(CASE WHEN newbalanceDest < 0 THEN 1 ELSE 0 END) as newbalanceDest
    FROM paysim
'''

get_fraud_transactions_query = '''
    SELECT * FROM paysim WHERE isFraud = 1
'''

confusion_matrix_query = '''
    SELECT
        isFraud,
        isFlaggedFraud,
        COUNT(*) as count
    FROM paysim
    GROUP BY isFraud, isFlaggedFraud
'''

draining_transaction_stats_query = '''
    SELECT
        SUM(CASE WHEN amount = oldbalanceOrg THEN 1 ELSE 0 END) as draining_count,
        COUNT(*) as total_count
    FROM paysim
'''

draining_behavior_by_type_query = '''
    SELECT
        type,
        isFraud,
        SUM(CASE WHEN amount = oldbalanceOrg THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as draining_percentage
    FROM paysim
    GROUP BY type, isFraud
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
    datetime('2025-10-01 00:00:00', '-' || step || ' hours') date,
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

fraud_flagging_analysis = '''
    SELECT
        isFraud,
        isFlaggedFraud
    FROM paysim
'''