-- PostgreSQL DDL для макета базы данных "team/modules"
-- Схема для управления командой разработчиков и модулями, подобными Transformers

CREATE TABLE IF NOT EXISTS public.team (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    area TEXT,
    experience_years INTEGER NOT NULL DEFAULT 0,
    email TEXT,
    github TEXT,
    location TEXT
);

COMMENT ON TABLE public.team IS 'Team members information with roles and experience';
COMMENT ON COLUMN public.team.id IS 'Unique identifier for team member';
COMMENT ON COLUMN public.team.name IS 'Full name of the team member';
COMMENT ON COLUMN public.team.role IS 'Role in the team: maintainer, core-dev, infra, docs, nlp, vision';
COMMENT ON COLUMN public.team.area IS 'Specialization area: NLP, Vision, Audio, Training, Docs, Utils';
COMMENT ON COLUMN public.team.experience_years IS 'Years of professional experience';
COMMENT ON COLUMN public.team.email IS 'Contact email address';
COMMENT ON COLUMN public.team.github IS 'GitHub username for collaboration';
COMMENT ON COLUMN public.team.location IS 'Geographical location or timezone';


CREATE TABLE IF NOT EXISTS public.modules (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    owner_id INTEGER NOT NULL REFERENCES public.team(id) ON DELETE RESTRICT
);

COMMENT ON TABLE public.modules IS 'Transformers-like modules with ownership information';
COMMENT ON COLUMN public.modules.id IS 'Unique identifier for module';
COMMENT ON COLUMN public.modules.name IS 'Full module name (e.g., transformers.trainer)';
COMMENT ON COLUMN public.modules.description IS 'Description of module functionality and purpose';
COMMENT ON COLUMN public.modules.owner_id IS 'Reference to the team member who owns this module';


-- Индексы для повышения производительности запросов
CREATE INDEX IF NOT EXISTS idx_team_role ON public.team(role);
CREATE INDEX IF NOT EXISTS idx_team_area ON public.team(area);
CREATE INDEX IF NOT EXISTS idx_team_experience ON public.team(experience_years);

CREATE INDEX IF NOT EXISTS idx_modules_owner ON public.modules(owner_id);
CREATE INDEX IF NOT EXISTS idx_modules_name_pattern ON public.modules(name text_pattern_ops);
