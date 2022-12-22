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
import TrainPage from './components/pages/TrainPage';

const App = () => {
  const [value, setValue] = useState(0);
  const [tweetTable , setTweetTable]=useState([{}])
  const [analyze,setAnalyze]=useState([{}])

  const routes = [
    { path: '/', component: <LandingPage /> },
    { path: '/train', component: <TrainPage /> },
    { path: '/tweets', component: <TweetsPage setAnalyze={setAnalyze} tweetTable={tweetTable} setTweetTable={setTweetTable} /> },
    { path: '/dashboard', component: <DashboardPage analyze={analyze} /> },
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
