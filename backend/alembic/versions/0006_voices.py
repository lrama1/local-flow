"""voices

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-14 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

import localflow.models.base
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "voices",
        sa.Column("created_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(length=26), nullable=False),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("adapter_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("native_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("base_voice_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("language", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("gender", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("params_json", sa.JSON(), nullable=False),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column("preview_path", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("preview_text", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adapter_key", "kind", "native_id"),
    )
    with op.batch_alter_table("voices", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_voices_kind"), ["kind"], unique=False)
        batch_op.create_index(batch_op.f("ix_voices_adapter_key"), ["adapter_key"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("voices", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_voices_adapter_key"))
        batch_op.drop_index(batch_op.f("ix_voices_kind"))
    op.drop_table("voices")
