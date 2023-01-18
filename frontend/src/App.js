import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import React, { useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import OurAppBar from './components/OurAppBar';
import theme from './components/Theme';
import Footer from './components/Footer';
import LandingPage from './components/pages/LandingPage';
import DashboardPage from './components/pages/DashboardPage';
import AboutPage from './components/pages/AboutPage';
import TweetsPage from './components/pages/TweetsPage';

const App = () => {
  const [value, setValue] = useState(0);
  const [analyze, setAnalyze] = useState([]);
  const [tweetTable, setTweetTable] = useState([{}]);
  const [task, setTask] = useState('');
  const [tweets, setTweets] = useState([]);
  const [dataset, setDataset] = useState('');

  const routes = [
    { path: '/', component: <LandingPage /> },
    {
      path: '/tweets',
      component: (
        <TweetsPage
          analyze={analyze}
          setAnalyze={setAnalyze}
          tweetTable={tweetTable}
          setTweetTable={setTweetTable}
          task={task}
          setTask={setTask}
          tweets={tweets}
          setTweets={setTweets}
          setDataset={setDataset}
        />
      ),
    },
    {
      path: '/dashboard',
      component: (
        <DashboardPage
          analyze={analyze}
          setValue={setValue}
          setTask={setTask}
          setTweets={setTweets}
          dataset={dataset}
        />
      ),
    },
    { path: '/about', component: <AboutPage /> },
  ];

  return (
    <Router>
      <ThemeProvider theme={theme}>
        <React.Fragment>
          <OurAppBar value={value} setValue={setValue} />
          <Routes>
            {routes.map((route, index) => (
              <Route
                key={`${index}${route}`}
                path={route.path}
                element={route.component}
                value={value}
                setValue={setValue}
              />
            ))}
          </Routes>
          <Footer />
        </React.Fragment>
      </ThemeProvider>
    </Router>
  );
};

export default App;
