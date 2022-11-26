import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import React, { Fragment } from 'react';
import OurAppBar from './components/OurAppBar'

const App = () => {
  return (
    <Router>
      <Fragment>
        <OurAppBar/>
        <Routes>
          <Route path="/" element={<div>Hello</div>}/>
        </Routes>
      </Fragment>
    </Router>
  );
};

export default App;
