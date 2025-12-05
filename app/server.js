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
