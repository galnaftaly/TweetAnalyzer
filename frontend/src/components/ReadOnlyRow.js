import React from "react";
import CreateIcon from "@mui/icons-material/Create";
import EditIcon from "@mui/icons-material/Edit";
import {
  Box,
  TableContainer,
  Button,
  Snackbar,
  Table,
  TableBody,
  TableHead,
  TableRow,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AddBoxIcon from "@mui/icons-material/AddBox";
import DoneIcon from "@mui/icons-material/Done";
import ClearIcon from "@mui/icons-material/Clear";

const ReadOnlyRow = ({ tweet, handleEditClick, handleDeleteClick }) => {
  return (
    <tr>
      <td>{tweet.tweetText}</td>

      <td>
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
      </td>
    </tr>
  );
};

export default ReadOnlyRow;
