-- Создание пользователя приложения (read-only) если не существует
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'nlq_user') THEN
        CREATE USER nlq_user WITH PASSWORD 'nlq_pass';
    END IF;
END
$$;

-- Базовые права
GRANT CONNECT ON DATABASE transformers TO nlq_user;
GRANT USAGE ON SCHEMA public TO nlq_user;

-- Права на чтение всех существующих и будущих таблиц
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO nlq_user;

-- Комментарий для документации
COMMENT ON ROLE nlq_user IS 'Read-only user for NLQ application';