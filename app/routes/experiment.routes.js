import express from 'express';
import multer from 'multer';
import { predictText, predictUrl, predictCsv } from '../config/flaskClient.js';
import { formatModels, topPrediction } from './helpers/modelFormatter.js';

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
  let modelNames = [];
  let topTag = '';

  if (!articleText) {
    error = 'Please provide article text.';
  } else {
    try {
      textResult = await predictText(articleText);
      if (textResult?.models) {
        textResult.models = formatModels(textResult.models);
        modelNames = textResult.models.map((m) => m.name);
        topTag = topPrediction(textResult.models);
      }
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
    modelNames,
    topTag,
    error,
  });
});

router.post('/url', async (req, res) => {
  const { article_url: articleUrl } = req.body;
  let urlResult;
  let error;
  let modelNames = [];
  let topTag = '';

  if (!articleUrl) {
    error = 'Please provide a URL.';
  } else {
    try {
      urlResult = await predictUrl(articleUrl);
      if (urlResult?.models) {
        urlResult.models = formatModels(urlResult.models);
        modelNames = urlResult.models.map((m) => m.name);
        topTag = topPrediction(urlResult.models);
      }
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
    modelNames,
    topTag,
    error,
  });
});

router.post('/csv', upload.single('csv_file'), async (req, res) => {
  let csvResult;
  let error;
  let modelNames = [];

  if (!req.file) {
    error = 'Please upload a CSV file.';
  } else {
    try {
      csvResult = await predictCsv(req.file.buffer, req.file.originalname);
      if (csvResult?.rows?.length) {
        const formattedRows = csvResult.rows.map((row) => {
          const formattedModels = formatModels(row.models || []);
          const modelMap = {};
          formattedModels.forEach((m) => (modelMap[m.name] = m));
          return { ...row, models: formattedModels, modelMap };
        });
        csvResult.rows = formattedRows;
        modelNames = formattedRows[0].models?.map((m) => m.name) || [];
      }
    } catch (err) {
      error = err.message;
    }
  }

  res.render('experiment', {
    pageTitle: PAGE_TITLE,
    title: PAGE_TITLE,
    activePage: 'experiment',
    csvResult,
    modelNames,
    error,
  });
});

export default router;
