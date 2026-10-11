"""collections

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-11 09:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

import localflow.models.base
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("created_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(length=26), nullable=False),
        sa.Column("project_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("collections", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_collections_project_id"), ["project_id"], unique=False)

    op.create_table(
        "collection_assets",
        sa.Column("created_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(length=26), nullable=False),
        sa.Column("collection_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("asset_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "asset_id"),
    )
    with op.batch_alter_table("collection_assets", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_collection_assets_collection_id"), ["collection_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_collection_assets_asset_id"), ["asset_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("collection_assets", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_collection_assets_asset_id"))
        batch_op.drop_index(batch_op.f("ix_collection_assets_collection_id"))
    op.drop_table("collection_assets")
    with op.batch_alter_table("collections", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_collections_project_id"))
    op.drop_table("collections")
