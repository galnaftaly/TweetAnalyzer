import React from "react";
import EditIcon from "@mui/icons-material/Edit";
import TableCell, { tableCellClasses } from "@mui/material/TableCell";
import {
  Button,
  TableRow,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";

const StyledTableCell = styled(TableCell)(({ theme }) => ({
    [`&.${tableCellClasses.head}`]: {
      backgroundColor: theme.palette.common.blue,
      color: theme.palette.common.white,
    },
    [`&.${tableCellClasses.body}`]: {
      fontSize: 14,
      border:"1pt solid black"
      
    },
  }));

const ReadOnlyRow = ({ tweet, handleEditClick, handleDeleteClick }) => {
  return (
    <TableRow sx={{width:"700",maxWidth:"700",minWidth:"700",border:"1pt solid black"}}>
      <StyledTableCell sx={{ overflow:"hidden" ,border:"1pt solid black"}}><div style={{overflowX:"auto",height:"50px",maxHeight:"100",minHeight:"100"}}>{tweet.tweetText}</div></StyledTableCell>

      <StyledTableCell sx={{border:"1pt solid black"}}>
        <div align="center" style={{ margin:"2"}}>
        <Button
          type="button"
           variant="contained"
          endIcon={<EditIcon />}
          onClick={(event) => handleEditClick(event, tweet)}
        >
          Edit
        </Button>
        <Button
          type="button"
          variant="contained"
          endIcon={<DeleteOutlineIcon />}
          onClick={() => handleDeleteClick(tweet.id)}
        >
          Delete
        </Button>
        </div>
      </StyledTableCell>
    </TableRow>
  );
};

export default ReadOnlyRow;
