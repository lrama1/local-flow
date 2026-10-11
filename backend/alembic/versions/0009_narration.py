"""narration: shots.narration and projects.narrator_voice_id

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-11 14:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("shots", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("narration", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="")
        )
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(sa.Column("narrator_voice_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_column("narrator_voice_id")
    with op.batch_alter_table("shots", schema=None) as batch_op:
        batch_op.drop_column("narration")
