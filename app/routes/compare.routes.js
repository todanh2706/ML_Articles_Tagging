import express from 'express';
import multer from 'multer';
import { evaluateText, evaluateCsv } from '../config/flaskClient.js';

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

const PAGE_TITLE = 'So sanh hieu suat';

router.get('/', (_req, res) => {
  res.render('compare', { pageTitle: PAGE_TITLE, title: PAGE_TITLE, activePage: 'compare' });
});

router.post('/text', async (req, res) => {
  const { compare_text: compareText, true_tag: trueTag } = req.body;
  let textEvaluation;
  let error;

  if (!compareText || !trueTag) {
    error = 'Please provide both text and ground-truth tag.';
  } else {
    try {
      textEvaluation = await evaluateText(compareText, trueTag);
    } catch (err) {
      error = err.message;
    }
  }

  res.render('compare', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'compare',
    compareText,
    trueTag,
    textEvaluation,
    error,
  });
});

router.post('/csv', upload.single('csv_file'), async (req, res) => {
  let csvEvaluation;
  let error;

  if (!req.file) {
    error = 'Please upload a CSV file.';
  } else {
    try {
      csvEvaluation = await evaluateCsv(req.file.buffer, req.file.originalname);
    } catch (err) {
      error = err.message;
    }
  }

  res.render('compare', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'compare',
    csvEvaluation,
    error,
  });
});

export default router;
