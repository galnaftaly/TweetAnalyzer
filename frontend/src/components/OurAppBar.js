import React, { useState } from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import {
  IconButton,
  Menu,
  Container,
  Typography,
  Button,
  Tooltip,
  MenuItem,
} from '@mui/material';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InsertCommentIcon from '@mui/icons-material/InsertComment';

const OurAppBar = () => {

  return (
    <AppBar position="static">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            noWrap
            component="a"
            href="/"
            sx={{
              mr: 2,
              display: { xs: 'none', md: 'flex' },
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            Tweet Analyzer
          </Typography>
          <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            <Button
              size="large"
              color="inherit"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              <SmartToyIcon />
              Train
            </Button>
            <Button
              size="large"
              color="inherit"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              <InsertCommentIcon />
              Insert Tweet
            </Button>
            <Button
              size="large"
              color="inherit"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              <TrendingUpIcon />
              Dashboard
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};
export default OurAppBar;
