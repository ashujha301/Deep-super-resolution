CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',

    target_platform VARCHAR(50) NOT NULL,
    upscale_factor INT NOT NULL,
    auto_optimize BOOLEAN NOT NULL DEFAULT TRUE,

    total_images INT NOT NULL DEFAULT 0,
    completed_images INT NOT NULL DEFAULT 0,
    failed_images INT NOT NULL DEFAULT 0,

    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    original_filename TEXT NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size_bytes BIGINT NOT NULL,

    raw_gcs_path TEXT NOT NULL,
    processed_gcs_path TEXT,

    width INT,
    height INT,

    status VARCHAR(50) NOT NULL DEFAULT 'queued',

    output_format VARCHAR(20),
    compression_ratio FLOAT,

    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_images_job_id ON images(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);