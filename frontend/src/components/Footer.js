import React from 'react';
import { Container, Typography } from '@mui/material';

const Footer = (props) => {
  const styles = {
    footer: (theme) => ({
      backgroundColor: theme.palette.common.blue,
      width: '100%',
      verticalAlign: 'bottom',
      mt: 'auto',
    }),
  };
  return (
    <footer>
      <Container sx={styles.footer} maxWidth={false} disableGutters>
        <Typography variant="h5" color="white">
          Copyright © Tweet Analyzer 2022. All rights reserved
        </Typography>
      </Container>
    </footer>
  );
};

export default Footer;
