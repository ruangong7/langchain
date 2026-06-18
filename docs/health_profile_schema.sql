CREATE TABLE IF NOT EXISTS user_health_profile (
    user_id BIGINT PRIMARY KEY,
    display_name VARCHAR(64) DEFAULT '',
    gender VARCHAR(16) DEFAULT '',
    age INT NULL,
    height_cm DECIMAL(6,2) NULL,
    weight_kg DECIMAL(6,2) NULL,
    is_pregnant TINYINT(1) NOT NULL DEFAULT 0,
    is_breastfeeding TINYINT(1) NOT NULL DEFAULT 0,
    conditions_text TEXT NULL,
    allergies_text TEXT NULL,
    notes TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_medications (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    drug_name VARCHAR(128) NOT NULL,
    dosage VARCHAR(64) DEFAULT '',
    purpose VARCHAR(128) DEFAULT '',
    frequency VARCHAR(64) DEFAULT '',
    times_per_day INT NULL,
    administration_time VARCHAR(64) DEFAULT '',
    start_date DATE NULL,
    end_date DATE NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_medications_user_id (user_id)
);
