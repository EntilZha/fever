# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Sqlite wikipedia db that stores protos by fever formatted wikipedia url."""
from typing import Text, List, Tuple
import os
from contextlib import contextmanager

from sqlalchemy import Column, Integer, String, create_engine, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm.scoping import ScopedSession

from serene.protos import fever_pb2


Base = declarative_base()  # pylint: disable=invalid-name


class WikiPage(Base):
    __tablename__ = "wikipedia"
    id = Column(Integer, primary_key=True)
    wikipedia_url = Column(String, index=True)
    proto = Column(LargeBinary)


SQLITE_PREFIX = "sqlite:///"


class WikiDatabase:
    """A wrapper around sqlite wikipedia database that returns proto pages."""

    def __init__(
        self,
        fs_path: str = "data/wiki_proto.sqlite3",
        ramfs_path: str = "/dev/shm/wiki_proto.sqlite3",
    ):
        """Constructor.
        Args:
        db_path: Path to db to read or write
        """
        self._fs_path: str = fs_path
        self._ramfs_path: str = ramfs_path
        if os.path.exists(ramfs_path):
            self._db_path = self._ramfs_path
        else:
            self._db_path = self._fs_path

        self._engine = create_engine(SQLITE_PREFIX + self._db_path)
        Base.metadata.bind = self._engine

    def create(self):
        Base.metadata.create_all(self._engine)

    def drop(self):
        Base.metadata.drop_all(self._engine)

    @classmethod
    def from_local(cls, db_path):
        db = cls(db_path)
        return db

    @property
    @contextmanager
    def session_scope(self) -> ScopedSession:
        session = scoped_session(sessionmaker(bind=self._engine))
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_wikipedia_urls(self):
        """Return a list of all wikipedia urls in this database.
        Returns:
        A list of wikipedia url strings in db
        """
        with self.session_scope as session:
            wiki_rows: List[Tuple[Text]] = session.query(WikiPage.wikipedia_url).all()
            return [row[0] for row in wiki_rows]

    def get_all_pages(self):
        with self.session_scope as session:
            for row in session.query(WikiPage):
                yield row.proto

    def get_page(self, wikipedia_url):
        """Get the proto for the wikipedia page by url.
        Args:
        wikipedia_url: Url to lookup
        Returns:
        Returns proto of page if it exists, otherwise None
        """
        wikipedia_url = wikipedia_url.replace("_", " ")
        with self.session_scope as session:
            page = (
                session.query(WikiPage).filter_by(wikipedia_url=wikipedia_url).first()
            )
            if page is None:
                return None
            else:
                return fever_pb2.WikipediaDump.FromString(page.proto)

    def get_page_sentence(self, wikipedia_url, sentence_id):
        """Get the text for a sentence on the given wikipedia page.
        Args:
        wikipedia_url: Url to lookup
        sentence_id: The sentence to retrieve
        Returns:
        The text of the sentence if it exists, None otherwise
        """
        wikipedia_url = wikipedia_url.replace("_", " ")
        with self.session_scope as session:
            page = (
                session.query(WikiPage).filter_by(wikipedia_url=wikipedia_url).first()
            )
            if page is None:
                return None
            else:
                page_proto = fever_pb2.WikipediaDump.FromString(page.proto)
                if sentence_id in page_proto.sentences:
                    return page_proto.sentences[sentence_id].text
                else:
                    return None
