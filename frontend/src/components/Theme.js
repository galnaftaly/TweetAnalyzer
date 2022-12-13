import { createTheme } from '@mui/material/styles';

//const ourBlue = '#0B72B9';
const ourBlue = '#2196f3';
const ourLightBlue = '#9fa8da';
const ourGrey = '#9e9e9e';

const theme = createTheme({
  palette: {
    common: {
      blue: `${ourBlue}`,
      lightBlue: `${ourLightBlue}`,
      grey: `${ourGrey}`,
    },
    primary: {
      main: `${ourBlue}`,
    },
    secondary: {
      main: `${ourLightBlue}`,
    },
  },
  typography: {
    fontSize: 12,
    tab: {
      fontWeight: 500,
      fontFamily: 'Raleway',
      fontSize: '1rem',
    },
    estimate: {
      fontFamily: 'Raleway',
      fontSize: '1rem',
      textTransform: 'none',
      color: 'white',
    },
    h2: {
      fontFamily: 'Raleway',
      fontWeight: 700,
      fontSize: '3.5rem',
      color: `${ourBlue}`,
      lineHeight: 1.25,
    },
    h3: {
      fontFamily: 'Pacifico',
      fontSize: '4rem',
      color: `${ourBlue}`,
    },
    h4: {
      fontFamily: 'Raleway',
      fontSize: '2.75rem',
      color: `${ourBlue}`,
      fontWeight: 700,
    },
    h5: {
      fontFamily: 'Arial',
      fontSize: 22,
      //lineHeight: 2,
      fontWeight: 600,
    },
    h6: {
      fontFamily: 'Raleway',
      fontWeight: 700,
      fontSize: '2.5rem',
      color: `${ourGrey}`,
      lineHeight: 1.25,
    },
  },
});

export default theme;
