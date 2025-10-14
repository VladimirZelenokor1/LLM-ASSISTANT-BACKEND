-- MOCK data themed to huggingface/transformers (fictional people)

-- team (github = вымышленный логин)
INSERT INTO public.team(name, role, area, experience_years, email, github, location) VALUES
('Ирина Петрова','maintainer','NLP',6,'irina@example.com','irina-maint','Moscow'),
('Алексей Смирнов','core-dev','NLP',4,'alexey@example.com','alex-core','Saint Petersburg'),
('Мария Кузнецова','core-dev','NLP',3,'maria@example.com','maria-core','Kazan'),
('Олег Иванов','infra','Training',5,'oleg@example.com','oleg-infra','Moscow'),
('Дмитрий Орлов','docs','Docs',2,'dmitry@example.com','dmitry-docs','Novosibirsk'),
('Светлана Сергеева','nlp','NLP',5,'svetlana@example.com','svet-nlp','Moscow'),
('Павел Волков','vision','Vision',4,'pavel@example.com','pavel-vision','Perm'),
('Анна Федорова','core-dev','Audio',3,'anna@example.com','anna-audio','Samara'),
('Григорий Литвинов','core-dev','Utils',2,'grigory@example.com','grig-utils','Tomsk')
ON CONFLICT DO NOTHING;

-- modules (owner_id берём по github-нику через подзапрос)
INSERT INTO public.modules(name, description, owner_id) VALUES
('transformers.pipelines','Pipelines registry and helpers',
 (SELECT id FROM public.team WHERE github='alex-core' LIMIT 1)),
('transformers.trainer','High-level training loop',
 (SELECT id FROM public.team WHERE github='oleg-infra' LIMIT 1)),
('transformers.tokenization','Tokenizer base utilities',
 (SELECT id FROM public.team WHERE github='svet-nlp' LIMIT 1)),
('transformers.models.gpt2','GPT-2 model package',
 (SELECT id FROM public.team WHERE github='svet-nlp' LIMIT 1)),
('transformers.models.bert','BERT model package',
 (SELECT id FROM public.team WHERE github='alex-core' LIMIT 1)),
('transformers.models.vit','Vision Transformer models',
 (SELECT id FROM public.team WHERE github='pavel-vision' LIMIT 1)),
('transformers.models.whisper','Whisper ASR models',
 (SELECT id FROM public.team WHERE github='anna-audio' LIMIT 1)),
('transformers.utils','Common helpers and adapters',
 (SELECT id FROM public.team WHERE github='grig-utils' LIMIT 1))
ON CONFLICT (name) DO NOTHING;

-- Обновление прав после создания таблиц
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nlq_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO nlq_user;
