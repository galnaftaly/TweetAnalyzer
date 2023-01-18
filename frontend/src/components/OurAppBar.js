import React, { useEffect } from 'react';
import { Typography, Tabs, Tab, Toolbar, AppBar } from '@mui/material';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TwitterIcon from '@mui/icons-material/Twitter';
import InfoIcon from '@mui/icons-material/Info';
import HomeIcon from '@mui/icons-material/Home';
import { Link } from 'react-router-dom';

const OurAppBar = (props) => {
  const routes = [
    {
      name: 'Home',
      link: '/',
      icon: <HomeIcon />,
    },
    {
      name: 'Insert Text',
      link: '/tweets',
      icon: <TwitterIcon />,
    },
    {
      name: 'Dashboard',
      link: '/dashboard',
      icon: <TrendingUpIcon />,
    },
    {
      name: 'About Us',
      link: '/about',
      icon: <InfoIcon />,
    },
  ];

  const styles = {
    toolbarMargin: (theme) => ({
      ...theme.mixins.toolbar,
      mb: '3em',
    }),
    tabContainer: {
      ml: 'auto',
    },
    tab: (theme) => ({
      fontSize: 15,
      fontFamily: 'Raleway',
      color: 'white',
      minWidth: 10,
      ml: '10px',
      '&:hover': {
        backgroundColor: theme.palette.primary.light,
      },
    }),
    typography: {
      m: 2,
      fontFamily: 'Raleway',
      fontWeight: 700,
      color: 'inherit',
      textDecoration: 'none',
    },
  };

  const handleChange = (event, newValue) => {
    props.setValue(newValue);
  };

  useEffect(() => {
    if (window.location.pathname === '/' && props.value !== 0) {
      props.setValue(0);
    } else if (window.location.pathname === '/tweets' && props.value !== 1) {
      props.setValue(1);
    } else if (window.location.pathname === '/dashboard' && props.value !== 2) {
      props.setValue(2);
    } else if (window.location.pathname === '/about' && props.value !== 3) {
      props.setValue(3);
    }
  }, [props, props.value]);

  return (
    <React.Fragment>
      <AppBar position="fixed">
        <Toolbar disableGutters>
          <Typography variant="h2" noWrap sx={styles.typography}>
            Text Analyzer
          </Typography>
          <Tabs
            sx={styles.tabContainer}
            value={props.value}
            onChange={handleChange}
            textColor="inherit"
            TabIndicatorProps={{
              style: { background: 'white', height: 2 },
            }}
          >
            {routes.map((route, index) => (
              <Tab
                key={`${route}${index}`}
                sx={styles.tab}
                component={Link}
                to={route.link}
                label={route.name}
                icon={route.icon}
                iconPosition="start"
              />
            ))}
          </Tabs>
        </Toolbar>
      </AppBar>
      <Toolbar sx={styles.toolbarMargin} />
    </React.Fragment>
  );
};
export default OurAppBar;
