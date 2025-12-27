import axios from 'axios';
import FormData from 'form-data';

const baseURL = process.env.FLASK_BASE_URL || 'http://localhost:5001';

const DEFAULT_TIMEOUT_MS = Number(process.env.FLASK_TIMEOUT_MS) || 10000;
const CRAWL_TIMEOUT_MS = Number(process.env.FLASK_CRAWL_TIMEOUT_MS) || 60000;

const flask = axios.create({
  baseURL,
  timeout: DEFAULT_TIMEOUT_MS,
});

function toError(e, fallbackMessage) {
  if (e.response?.data?.error) {
    return new Error(e.response.data.error);
  }
  if (e.response?.data?.message) {
    return new Error(e.response.data.message);
  }
  return new Error(fallbackMessage || 'Flask service call failed');
}

export async function predictText(text) {
  try {
    const { data } = await flask.post('/predict/text', { text });
    return data;
  } catch (e) {
    throw toError(e, 'Unable to predict from text');
  }
}

export async function predictUrl(url) {
  try {
    const { data } = await flask.post('/predict/url', { url });
    return data;
  } catch (e) {
    throw toError(e, 'Unable to predict from URL');
  }
}

export async function predictCsv(fileBuffer, filename) {
  try {
    const form = new FormData();
    form.append('file', fileBuffer, {
      filename: filename || 'upload.csv',
      contentType: 'text/csv',
    });

    const { data } = await flask.post('/predict/csv', form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });
    return data;
  } catch (e) {
    throw toError(e, 'Unable to predict from CSV');
  }
}

export async function evaluateText(text, trueTag) {
  try {
    const { data } = await flask.post('/evaluate/text', { text, true_tag: trueTag });
    return data;
  } catch (e) {
    throw toError(e, 'Unable to evaluate text');
  }
}

export async function evaluateCsv(fileBuffer, filename) {
  try {
    const form = new FormData();
    form.append('file', fileBuffer, {
      filename: filename || 'dataset.csv',
      contentType: 'text/csv',
    });

    const { data } = await flask.post('/evaluate/csv', form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });
    return data;
  } catch (e) {
    throw toError(e, 'Unable to evaluate CSV');
  }
}

export async function crawlTag(tag) {
  try {
    const { data } = await flask.get('/crawl/tag', { params: { tag }, timeout: CRAWL_TIMEOUT_MS });
    return data;
  } catch (e) {
    throw toError(e, 'Unable to crawl articles for tag');
  }
}

export default {
  predictText,
  predictUrl,
  predictCsv,
  evaluateText,
  evaluateCsv,
  crawlTag,
};
