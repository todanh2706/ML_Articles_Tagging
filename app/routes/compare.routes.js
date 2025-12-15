import express from 'express';
import multer from 'multer';
import { evaluateText, evaluateCsv } from '../config/flaskClient.js';
import { formatModels } from './helpers/modelFormatter.js';

function normalizeTag(value = '') {
  const cleaned = value
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned;
}

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
  let modelNames = [];
  const normalizedTrueTag = normalizeTag(trueTag);

  if (!compareText || !trueTag) {
    error = 'Please provide both text and ground-truth tag.';
  } else {
    try {
      textEvaluation = await evaluateText(compareText, normalizedTrueTag);
      if (textEvaluation?.models) {
        textEvaluation.models = formatModels(textEvaluation.models);
        modelNames = textEvaluation.models.map((m) => m.name);
      }
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
    modelNames,
    error,
  });
});

router.post('/csv', upload.single('csv_file'), async (req, res) => {
  let csvEvaluation;
  let error;
  let modelNames = [];

  if (!req.file) {
    error = 'Please upload a CSV file.';
  } else {
    try {
      csvEvaluation = await evaluateCsv(req.file.buffer, req.file.originalname);
      if (csvEvaluation?.summary) {
        modelNames = Object.keys(csvEvaluation.summary);
      }
      if (csvEvaluation?.rows?.length && modelNames.length === 0) {
        modelNames = csvEvaluation.rows[0].models?.map((m) => m.name) || [];
      }
      if (csvEvaluation?.rows?.length) {
        csvEvaluation.rows = csvEvaluation.rows.map((row) => {
          const modelMap = {};
          const formatted = formatModels(row.models || []);
          formatted.forEach((m) => (modelMap[m.name] = m));
          return { ...row, modelMap };
        });
      }
    } catch (err) {
      error = err.message;
    }
  }

  res.render('compare', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'compare',
    csvEvaluation,
    modelNames,
    error,
  });
});

export default router;
