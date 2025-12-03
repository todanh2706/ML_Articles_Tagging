import express from 'express';
import multer from 'multer';
import { predictText, predictUrl, predictCsv } from '../config/flaskClient.js';

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === 'text/csv' || file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  },
});

const PAGE_TITLE = 'Thuc nghiem mo hinh';

router.get('/', (_req, res) => {
  res.render('experiment', { pageTitle: PAGE_TITLE, title: PAGE_TITLE, activePage: 'experiment' });
});

router.post('/text', async (req, res) => {
  const { article_text: articleText } = req.body;
  let textResult;
  let error;

  if (!articleText) {
    error = 'Please provide article text.';
  } else {
    try {
      textResult = await predictText(articleText);
    } catch (err) {
      error = err.message;
    }
  }

  res.render('experiment', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'experiment',
    inputText: articleText,
    textResult,
    error,
  });
});

router.post('/url', async (req, res) => {
  const { article_url: articleUrl } = req.body;
  let urlResult;
  let error;

  if (!articleUrl) {
    error = 'Please provide a URL.';
  } else {
    try {
      urlResult = await predictUrl(articleUrl);
    } catch (err) {
      error = err.message;
    }
  }

  res.render('experiment', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'experiment',
    urlInput: articleUrl,
    urlResult,
    error,
  });
});

router.post('/csv', upload.single('csv_file'), async (req, res) => {
  let csvResult;
  let error;

  if (!req.file) {
    error = 'Please upload a CSV file.';
  } else {
    try {
      csvResult = await predictCsv(req.file.buffer, req.file.originalname);
    } catch (err) {
      error = err.message;
    }
  }

  res.render('experiment', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'experiment',
    csvResult,
    error,
  });
});

export default router;
