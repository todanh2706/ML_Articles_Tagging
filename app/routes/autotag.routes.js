import express from 'express';
import { crawlTag } from '../config/flaskClient.js';

const router = express.Router();

const TAGS = [
  { key: 'chinh-tri', label: 'Chính trị' },
  { key: 'kinh-te', label: 'Kinh tế' },
  { key: 'giao-duc', label: 'Giáo dục' },
  { key: 'the-thao', label: 'Thể thao' },
  { key: 'giai-tri', label: 'Giải trí' },
  { key: 'cong-nghe', label: 'Công nghệ' },
  { key: 'doi-song', label: 'Đời sống' },
];

const PAGE_TITLE = 'Gán tag tự động';

router.get('/', (_req, res) => {
  res.render('autotag', { pageTitle: PAGE_TITLE, title: PAGE_TITLE, tags: TAGS, activePage: 'autotag' });
});

router.get('/tag/:tagName', async (req, res) => {
  const { tagName } = req.params;
  const { sort } = req.query;

  let articles = [];
  let error;

  try {
    const data = await crawlTag(tagName);
    articles = data?.articles || data || [];
  } catch (err) {
    error = err.message;
  }

  if (sort === 'confidence') {
    articles = [...articles].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  } else {
    articles = [...articles].sort((a, b) => new Date(b.published_at || 0) - new Date(a.published_at || 0));
  }

  res.render('autotag', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    tags: TAGS,
    selectedTag: tagName,
    sortOption: sort || 'newest',
    articles,
    activePage: 'autotag',
    error,
  });
});

export default router;
