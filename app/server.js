import express from 'express';
import { engine } from 'express-handlebars';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

import experimentRoutes from './routes/experiment.routes.js';
import compareRoutes from './routes/compare.routes.js';
import autotagRoutes from './routes/autotag.routes.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.engine(
  'hbs',
  engine({
    extname: '.hbs',
    defaultLayout: 'main',
    layoutsDir: path.join(__dirname, 'views', 'layouts'),
    partialsDir: path.join(__dirname, 'views', 'partials'),
    helpers: {
      ifEquals(a, b, options) {
        return a === b ? options.fn(this) : options.inverse(this);
      },
      unlessEquals(a, b, options) {
        return a !== b ? options.fn(this) : options.inverse(this);
      },
      eq(a, b) {
        return a === b;
      },
      formatDateTime(value) {
        if (!value) return '';

        let s = String(value);
        // cắt phần microseconds .462870 cho chắc ăn
        if (s.includes('.')) {
          s = s.split('.')[0]; // "2025-12-10T21:09:56"
        }

        const date = new Date(s);
        if (isNaN(date.getTime())) {
          return value; // parse không được thì trả nguyên
        }

        const pad = (n) => String(n).padStart(2, '0');

        const day = pad(date.getDate());
        const month = pad(date.getMonth() + 1);
        const year = date.getFullYear();
        const hour = pad(date.getHours());
        const minute = pad(date.getMinutes());
        const second = pad(date.getSeconds());

        return `${day}/${month}/${year} ${hour}:${minute}:${second}`;
      },
    },
  })
);
app.set('view engine', 'hbs');
app.set('views', path.join(__dirname, 'views'));

app.use(express.json({ limit: '2mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (_req, res) => res.redirect('/experiment'));
app.use('/experiment', experimentRoutes);
app.use('/compare', compareRoutes);
app.use('/autotag', autotagRoutes);

app.use((req, res) => {
  res.status(404).render('experiment', {
    pageTitle: 'Page not found',
    error: 'The page you are looking for does not exist.',
  });
});

app.use((err, req, res, _next) => {
  console.error('Unhandled error:', err);
  const status = err.status || 500;
  res.status(status).render('experiment', {
    pageTitle: 'Server error',
    error: err.message || 'An unexpected error occurred. Please try again.',
  });
});

app.listen(PORT, () => {
  console.log(`Express server listening at http://localhost:${PORT}`);
});
