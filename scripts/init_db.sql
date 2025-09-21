-- Multi-Agent City Database Initialization Script

-- Create database if it doesn't exist
-- (This is handled by Docker environment variables)

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    user_id UUID NOT NULL,
    profile JSONB NOT NULL,
    state JSONB NOT NULL,
    model_config JSONB NOT NULL,
    external_endpoint VARCHAR(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create events table
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL,
    agent_id UUID,
    data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    district VARCHAR(100),
    processed BOOLEAN DEFAULT FALSE
);

-- Create districts table
CREATE TABLE IF NOT EXISTS districts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    neighborhoods JSONB NOT NULL,
    population INTEGER DEFAULT 0,
    governance JSONB NOT NULL,
    economy JSONB NOT NULL
);

-- Create properties table
CREATE TABLE IF NOT EXISTS properties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL,
    district VARCHAR(100) NOT NULL,
    neighborhood VARCHAR(100) NOT NULL,
    address VARCHAR(255) NOT NULL,
    owner_id UUID,
    lease_terms JSONB,
    features JSONB DEFAULT '[]'::jsonb,
    size FLOAT NOT NULL,
    price FLOAT NOT NULL
);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sender_id UUID NOT NULL,
    receiver_id UUID NOT NULL,
    amount FLOAT NOT NULL,
    currency VARCHAR(20) DEFAULT 'credits',
    type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending'
);

-- Create memories table
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    importance FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    participants JSONB DEFAULT '[]'::jsonb,
    emotions JSONB DEFAULT '[]'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents USING GIN ((state->>'status'));
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_agent_id ON events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_district ON events(district);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_properties_district ON properties(district);
CREATE INDEX IF NOT EXISTS idx_properties_owner ON properties(owner_id);
CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender_id);
CREATE INDEX IF NOT EXISTS idx_transactions_receiver ON transactions(receiver_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

-- Create GIN indexes for JSONB fields
CREATE INDEX IF NOT EXISTS idx_agents_profile ON agents USING GIN (profile);
CREATE INDEX IF NOT EXISTS idx_agents_state ON agents USING GIN (state);
CREATE INDEX IF NOT EXISTS idx_events_data ON events USING GIN (data);
CREATE INDEX IF NOT EXISTS idx_properties_features ON properties USING GIN (features);
CREATE INDEX IF NOT EXISTS idx_transactions_metadata ON transactions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_memories_participants ON memories USING GIN (participants);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags);

-- Insert default districts
INSERT INTO districts (name, neighborhoods, population, governance, economy) VALUES
(
    'Downtown',
    '["Financial District", "Arts Quarter", "Government Center"]'::jsonb,
    0,
    '{"council_size": 5, "voting_threshold": 0.6}'::jsonb,
    '{"base_rent": 100, "business_tax": 0.1}'::jsonb
) ON CONFLICT (name) DO NOTHING;

INSERT INTO districts (name, neighborhoods, population, governance, economy) VALUES
(
    'Market Square',
    '["Central Market", "Food Court", "Artisan Row"]'::jsonb,
    0,
    '{"council_size": 3, "voting_threshold": 0.5}'::jsonb,
    '{"base_rent": 75, "business_tax": 0.08}'::jsonb
) ON CONFLICT (name) DO NOTHING;

INSERT INTO districts (name, neighborhoods, population, governance, economy) VALUES
(
    'Tech Hub',
    '["Innovation Campus", "Startup District", "Research Park"]'::jsonb,
    0,
    '{"council_size": 7, "voting_threshold": 0.7}'::jsonb,
    '{"base_rent": 150, "business_tax": 0.05}'::jsonb
) ON CONFLICT (name) DO NOTHING;

INSERT INTO districts (name, neighborhoods, population, governance, economy) VALUES
(
    'Residential',
    '["Green Hills", "Riverside", "Old Town"]'::jsonb,
    0,
    '{"council_size": 4, "voting_threshold": 0.55}'::jsonb,
    '{"base_rent": 50, "business_tax": 0.12}'::jsonb
) ON CONFLICT (name) DO NOTHING;

-- Insert sample properties
INSERT INTO properties (type, district, neighborhood, address, features, size, price) VALUES
('bakery', 'Market Square', 'Central Market', '123 Baker Street', '["kitchen", "storefront", "storage"]'::jsonb, 150.0, 75.0),
('cafe', 'Market Square', 'Food Court', '456 Coffee Lane', '["kitchen", "seating", "wifi"]'::jsonb, 100.0, 60.0),
('office', 'Tech Hub', 'Startup District', '789 Innovation Ave', '["open_space", "meeting_rooms", "high_speed_internet"]'::jsonb, 200.0, 120.0),
('apartment', 'Residential', 'Green Hills', '321 Peaceful St', '["bedroom", "kitchen", "balcony"]'::jsonb, 75.0, 40.0),
('shop', 'Downtown', 'Arts Quarter', '654 Creative Blvd', '["display_area", "storage", "workshop"]'::jsonb, 120.0, 85.0),
('restaurant', 'Market Square', 'Food Court', '987 Tasty Way', '["full_kitchen", "dining_area", "bar"]'::jsonb, 180.0, 90.0),
('studio', 'Residential', 'Riverside', '147 Quiet Lane', '["living_space", "workspace"]'::jsonb, 50.0, 35.0),
('coworking', 'Tech Hub', 'Innovation Campus', '258 Collaborate St', '["desks", "meeting_rooms", "cafe"]'::jsonb, 300.0, 150.0)
ON CONFLICT DO NOTHING;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean old events
CREATE OR REPLACE FUNCTION clean_old_events()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM events
    WHERE processed = TRUE
    AND timestamp < NOW() - INTERVAL '7 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get agent statistics
CREATE OR REPLACE FUNCTION get_agent_stats()
RETURNS TABLE(
    total_agents INTEGER,
    active_agents INTEGER,
    by_district JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_agents,
        COUNT(CASE WHEN state->>'status' = 'active' THEN 1 END)::INTEGER as active_agents,
        jsonb_object_agg(
            state->>'location'->>'district',
            COUNT(*)
        ) as by_district
    FROM agents
    WHERE state->>'location' IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- Create admin user for development (remove in production)
-- CREATE USER admin_user WITH PASSWORD 'admin_password';
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin_user;

COMMIT;