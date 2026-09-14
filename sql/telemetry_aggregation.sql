-- PostgreSQL. Input: game_events(player_id, event_name, event_timestamp,
-- event_value_usd) and experiment_assignments(player_id, variant,
-- assigned_at, platform, country, acquisition_channel).
-- Produces leakage-auditable player-level telemetry relative to assignment.
WITH events AS (
    SELECT
        a.player_id,
        a.variant,
        a.assigned_at,
        a.platform,
        a.country,
        a.acquisition_channel,
        e.event_name,
        e.event_timestamp,
        COALESCE(e.event_value_usd, 0.0) AS event_value_usd,
        EXTRACT(EPOCH FROM (e.event_timestamp - a.assigned_at)) / 86400.0 AS day_since_assignment
    FROM experiment_assignments AS a
    LEFT JOIN game_events AS e
      ON e.player_id = a.player_id
     AND e.event_timestamp >= a.assigned_at
     AND e.event_timestamp < a.assigned_at + INTERVAL '8 days'
), player_metrics AS (
    SELECT
        player_id,
        variant,
        MIN(assigned_at) AS assigned_at,
        MIN(platform) AS platform,
        MIN(country) AS country,
        MIN(acquisition_channel) AS acquisition_channel,
        COUNT(*) FILTER (WHERE event_name = 'session_start' AND day_since_assignment < 1) AS sessions_d1,
        COALESCE(SUM(event_value_usd) FILTER (WHERE event_name = 'purchase' AND day_since_assignment < 1), 0) AS revenue_d1,
        COALESCE(SUM(event_value_usd) FILTER (WHERE event_name = 'purchase' AND day_since_assignment < 8), 0) AS revenue_d7,
        (COUNT(*) FILTER (WHERE event_name = 'session_start' AND day_since_assignment >= 1 AND day_since_assignment < 2) > 0)::int AS retained_d1,
        (COUNT(*) FILTER (WHERE event_name = 'session_start' AND day_since_assignment >= 7 AND day_since_assignment < 8) > 0)::int AS retained_d7
    FROM events
    GROUP BY player_id, variant
)
SELECT *, 1 - retained_d7 AS churned_d7
FROM player_metrics;
