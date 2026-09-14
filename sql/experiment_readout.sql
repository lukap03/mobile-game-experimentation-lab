-- Input: player_metrics, produced by telemetry_aggregation.sql.
WITH variant_metrics AS (
    SELECT
        variant,
        COUNT(*) AS players,
        AVG(retained_d7::numeric) AS d7_retention,
        AVG(retained_d1::numeric) AS d1_retention,
        AVG(sessions_d1::numeric) AS mean_sessions_d1,
        AVG(revenue_d7::numeric) AS arpu_d7
    FROM player_metrics
    GROUP BY variant
), pivoted AS (
    SELECT
        MAX(d7_retention) FILTER (WHERE variant = 'treatment') - MAX(d7_retention) FILTER (WHERE variant = 'control') AS d7_absolute_uplift,
        MAX(d1_retention) FILTER (WHERE variant = 'treatment') - MAX(d1_retention) FILTER (WHERE variant = 'control') AS d1_absolute_uplift,
        MAX(mean_sessions_d1) FILTER (WHERE variant = 'treatment') - MAX(mean_sessions_d1) FILTER (WHERE variant = 'control') AS sessions_absolute_uplift,
        MAX(arpu_d7) FILTER (WHERE variant = 'treatment') - MAX(arpu_d7) FILTER (WHERE variant = 'control') AS arpu_absolute_uplift
    FROM variant_metrics
)
SELECT * FROM variant_metrics
UNION ALL
SELECT 'treatment_minus_control', NULL, d7_absolute_uplift, d1_absolute_uplift, sessions_absolute_uplift, arpu_absolute_uplift
FROM pivoted;
