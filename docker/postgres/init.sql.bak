-- Create user if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
        CREATE USER airflow WITH PASSWORD 'airflow';
    END IF;
END $$;

-- Create database if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prod_db') THEN
        CREATE DATABASE prod_db;
    END IF;
END $$;

-- Grant privileges to airflow user
GRANT ALL PRIVILEGES ON DATABASE prod_db TO airflow;

-- Ensure airflow owns the public schema in prod_db
ALTER SCHEMA public OWNER TO airflow;
GRANT ALL ON SCHEMA public TO airflow;