from contextlib import contextmanager
import typing

from sqlalchemy import (
    Engine,
    ForeignKey,
    MetaData,
    Table,
    Column,
    Integer,
    Text,
    JSON,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.sql import expression as expr
from sqlalchemy.exc import SQLAlchemyError
from tklearn.core.model import Document, Source

from tklearn.datasets.stores.base import DataStore, register
from tklearn.utils import logging

__all__ = [
    'sa_metadata_obj',
    'document_table',
    'SQLAlchemyDataStore',
]

logger = logging.get_logger(__name__)

sa_metadata_obj = MetaData()

source_table = Table(
    'source',
    sa_metadata_obj,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', Text, nullable=False),
    UniqueConstraint('name', name='uc_name'),
)

document_table = Table(
    'document',
    sa_metadata_obj,
    Column('_id', Integer, primary_key=True, autoincrement=True),
    Column('source_id', Integer, ForeignKey('source.id'), nullable=False),
    Column('id', Text, nullable=False),
    Column('data', JSON),
    UniqueConstraint('source_id', 'id', name='uc_source_doc_id')
)


@register('sqlalchemy')
class SQLAlchemyDataStore(DataStore):
    def create_all(self):
        sa_metadata_obj.create_all(self._engine)

    @property
    def _engine(self) -> Engine:
        if not hasattr(self, '_engine_') or self._engine_ is None:
            uri = self.env.format(self.args.get(
                'uri', self.env.config['SQLALCHEMY_URI']
            ))
            print(uri)
            self._engine_ = create_engine(uri)
            self.create_all()
        return self._engine_

    @contextmanager
    def connect(self):
        with self._engine.connect() as conn:
            yield conn

    def filter_source_by(self, name: str = None, id: int = None, create=True) \
            -> Source:
        """Get or create a source with the given name.

        Parameters
        ----------
        name : str
            The name of the source.
        create : bool
            Whether to create the source if it does not exist.

        Returns
        -------
        Source
            The source object.
        """
        with self.connect() as conn:
            if create and name is not None:
                # try to create source
                stmt = expr.insert(source_table).values(name=name)
                try:
                    conn.execute(stmt)
                    conn.commit()
                except SQLAlchemyError as e:
                    logger.error('Error adding source: %s', e)
                    conn.rollback()
            # get source
            stmt = expr.select(source_table)
            if id is not None:
                stmt = stmt.where(source_table.c.id == id)
            if name is not None:
                stmt = stmt.where(source_table.c.name == name)
            result = conn.execute(stmt).first()
        return Source.from_orm(result)

    def add(self, doc: typing.Union[Document, typing.Iterable[Document]], ignore_duplicates=True):
        # get or create source
        if isinstance(doc, Document):
            doc = [doc]
        with self.connect() as conn:
            for d in doc:
                if d.source.id is None:
                    source_name = d.source.name
                    if source_name is None:
                        source_name = self.args['default_source_name']
                    source = self.filter_source_by(name=source_name)
                    d.source.id = source.id
                    d.source.name = source.name
                if not isinstance(d, Document):
                    raise TypeError(
                        'doc must be an instance of Document or '
                        'an iterable of Document instances.'
                    )
                stmt = expr.insert(document_table).values(
                    source_id=d.source.id,
                    id=d.id,
                    data=d.data
                )
                try:
                    _ = conn.execute(stmt)
                except SQLAlchemyError as e:
                    conn.rollback()
                    UNIQUE_CONSTRAINT_ERROR = 'UNIQUE constraint failed: document.source_id, document.id'
                    if not ignore_duplicates and UNIQUE_CONSTRAINT_ERROR in str(e):
                        logger.error('Error adding document: %s', e.args)
                        raise e
            try:
                conn.commit()
            except SQLAlchemyError as e:
                logger.error('Error adding document: %s', e)
                conn.rollback()
                raise e

    def delete(self, source: Source):
        """Delete the source along with all documents from the store.

        Parameters
        ----------
        source : Source
            Source of the documents to delete.

        Returns
        None
            None
        """
        if not isinstance(source, Source):
            raise TypeError('source must be an instance of Source.')
        with self.connect() as conn:
            if source.id is not None:
                stmt = expr.delete(source_table).where(
                    source_table.c.id == source.id
                )
            else:
                stmt = expr.delete(source_table).where(
                    source_table.c.name == source.name
                )
            try:
                conn.execute(stmt)
                conn.commit()
            except SQLAlchemyError as e:
                logger.error('Error deleting source: %s', e)
                conn.rollback()
                raise e

    def get(self, id: int) -> Document:
        """Get a document from the store by its id.

        Parameters
        ----------
        id : int
            Id of the document to get.

        Returns
        -------
        Document
            Document with the given id.
        """
        with self.connect() as conn:
            stmt = expr.select(document_table).where(
                document_table.c.id == id
            )
            doc = conn.scalars(stmt).first()
        source = self.filter_source_by(id=doc.source_id)
        doc = Document.from_orm(doc)
        doc.source = source
        return doc

    def count(self) -> int:
        """Count the number of documents in the store.

        Returns
        -------
        int
            Number of documents in the store.
        """
        with self.connect() as conn:
            stmt = expr.select(expr.func.count()).select_from(document_table)
            return conn.execute(stmt).scalar()

    def __iter__(self) -> typing.Iterator[Document]:
        stmt = expr.select(document_table)
        with self.connect() as conn:
            docs = conn.execute(stmt)
        for doc in docs:
            source = self.filter_source_by(id=doc.source_id)
            doc = Document.from_orm(doc)
            doc.source = source
            yield doc
