-- Таблица команды
CREATE TABLE IF NOT EXISTS public.team (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,                  -- maintainer | core-dev | infra | docs | nlp | vision
  area TEXT,                           -- NLP | Vision | Audio | Training | Docs | Utils ...
  experience_years INTEGER NOT NULL DEFAULT 0,
  email TEXT,
  github TEXT,                         -- вымышленный логин, для ссылок
  location TEXT
);

-- Таблица модулей
CREATE TABLE IF NOT EXISTS public.modules (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,           -- e.g. transformers.trainer
  description TEXT,
  owner_id INTEGER NOT NULL REFERENCES public.team(id)
);

-- Индексы для производительности
CREATE INDEX IF NOT EXISTS idx_team_role ON public.team(role);
CREATE INDEX IF NOT EXISTS idx_modules_owner ON public.modules(owner_id);
CREATE INDEX IF NOT EXISTS idx_modules_name ON public.modules(name);

-- Права для пользователя приложения
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nlq_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO nlq_user;